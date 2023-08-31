/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lammpsgui.h"

#include "chartviewer.h"
#include "highlighter.h"
#include "imageviewer.h"
#include "lammpsrunner.h"
#include "logwindow.h"
#include "preferences.h"
#include "setvariables.h"
#include "stdcapture.h"
#include "ui_lammpsgui.h"

#include <QClipboard>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QGuiApplication>
#include <QLabel>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QShortcut>
#include <QStatusBar>
#include <QStringList>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QUrl>

#include <cstring>
#include <string>

#if defined(_OPENMP)
#include <cstdlib>
#include <omp.h>
#endif

static const QString blank(" ");
static constexpr int MAXRECENT = 5;

// duplicate string
static char *mystrdup(const std::string &text)
{
    auto tmp = new char[text.size() + 1];
    memcpy(tmp, text.c_str(), text.size() + 1);
    return tmp;
}

// find if executable is in path
// https://stackoverflow.com/a/51041497

static bool has_exe(const QString &exe)
{
    QProcess findProcess;
    QStringList arguments;
    arguments << exe;
#if defined(_WIN32)
    findProcess.start("where", arguments);
#else
    findProcess.start("which", arguments);
#endif
    findProcess.setReadChannel(QProcess::ProcessChannel::StandardOutput);

    if (!findProcess.waitForFinished()) return false; // Not found or which does not work

    QString retStr(findProcess.readAll());
    retStr = retStr.trimmed();

    QFile file(retStr);
    QFileInfo check_file(file);
    if (check_file.exists() && check_file.isFile())
        return true; // Found!
    else
        return false; // Not found!
}

LammpsGui::LammpsGui(QWidget *parent, const char *filename) :
    QMainWindow(parent), ui(new Ui::LammpsGui), highlighter(nullptr), capturer(nullptr),
    status(nullptr), logwindow(nullptr), imagewindow(nullptr), chartwindow(nullptr),
    logupdater(nullptr), dirstatus(nullptr), progress(nullptr), prefdialog(nullptr),
    lammpsstatus(nullptr), varwindow(nullptr)
{
    // enforce using the plain ASCII C locale within the GUI.
    QLocale::setDefault(QLocale("C"));

    // register QList<QString>
    qRegisterMetaTypeStreamOperators<QList<QString>>("QList<QString>");

    ui->setupUi(this);
    this->setCentralWidget(ui->textEdit);
    highlighter = new Highlighter(ui->textEdit->document());
    capturer    = new StdCapture;
    current_file.clear();
    current_dir = QDir(".").absolutePath();

    QCoreApplication::setOrganizationName("The LAMMPS Developers");
    QCoreApplication::setOrganizationDomain("lammps.org");
    QCoreApplication::setApplicationName("LAMMPS GUI");

    // restore and initialize settings
    QSettings settings;

#if defined(LAMMPS_GUI_USE_PLUGIN)
    plugin_path.clear();
    std::string deffile = settings.value("plugin_path", "liblammps.so").toString().toStdString();
    for (const char *libfile : {deffile.c_str(), "./liblammps.so", "liblammps.dylib",
                                "./liblammps.dylib", "liblammps.dll"}) {
        if (lammps.load_lib(libfile)) {
            auto canonical = QFileInfo(libfile).canonicalFilePath();
            plugin_path    = canonical.toStdString();
            settings.setValue("plugin_path", canonical);
            break;
        }
    }

    if (plugin_path.empty()) {
        // none of the plugin paths could load, remove key
        settings.remove("plugin_path");
        QMessageBox::critical(this, "Error", "Cannot open LAMMPS shared library file");
        exit(1);
    }
#endif

    // switch configured accelerator back to "none" if needed.
    int accel = settings.value("accelerator", AcceleratorTab::None).toInt();
    if (accel == AcceleratorTab::Opt) {
        if (!lammps.config_has_package("OPT"))
            settings.setValue("accelerator", AcceleratorTab::None);
    } else if (accel == AcceleratorTab::OpenMP) {
        if (!lammps.config_has_package("OPENMP"))
            settings.setValue("accelerator", AcceleratorTab::None);
    } else if (accel == AcceleratorTab::Intel) {
        if (!lammps.config_has_package("INTEL"))
            settings.setValue("accelerator", AcceleratorTab::None);
    } else if (accel == AcceleratorTab::Gpu) {
        if (!lammps.config_has_package("GPU") || !lammps.has_gpu_device())
            settings.setValue("accelerator", AcceleratorTab::None);
    } else if (accel == AcceleratorTab::Kokkos) {
        if (!lammps.config_has_package("KOKKOS"))
            settings.setValue("accelerator", AcceleratorTab::None);
    }

    // check and initialize nthreads setting. Default is to use max if there
    // is no preference but do not override OMP_NUM_THREADS
#if defined(_OPENMP)
    // use maximum number of available threads unless OMP_NUM_THREADS was set
    int nthreads = settings.value("nthreads", omp_get_max_threads()).toInt();
    if (!qEnvironmentVariableIsSet("OMP_NUM_THREADS")) {
        qputenv("OMP_NUM_THREADS", std::to_string(nthreads).c_str());
    }
#else
    int nthreads = settings.value("nthreads", 1).toInt();
#endif
    settings.setValue("nthreads", QString::number(nthreads));

    lammps_args.clear();
    lammps_args.push_back(mystrdup("LAMMPS-GUI"));
    lammps_args.push_back(mystrdup("-log"));
    lammps_args.push_back(mystrdup("none"));

    setWindowIcon(QIcon(":/lammps-icon-128x128.png"));

    QFont all_font("Arial", -1);
    all_font.setStyleHint(QFont::SansSerif, QFont::PreferOutline);
    all_font.fromString(settings.value("allfont", all_font.toString()).toString());
    settings.setValue("allfont", all_font.toString());
    QApplication::setFont(all_font);

    QFont text_font("Monospace", -1);
    text_font.setStyleHint(QFont::Monospace, QFont::PreferOutline);
    text_font.fromString(settings.value("textfont", text_font.toString()).toString());
    settings.setValue("textfont", text_font.toString());
    ui->textEdit->setFont(text_font);
    ui->textEdit->setMinimumSize(600, 400);

    varwindow = new QLabel(QString());
    varwindow->setWindowTitle("LAMMPS-GUI - Current Variables:");
    varwindow->setWindowIcon(QIcon(":/lammps-icon-128x128.png"));
    varwindow->setMinimumSize(100, 50);
    varwindow->setText("(none)");
    varwindow->setFont(text_font);
    varwindow->setFrameStyle(QFrame::Sunken);
    varwindow->setFrameShape(QFrame::Panel);
    varwindow->setAlignment(Qt::AlignVCenter);
    varwindow->setContentsMargins(5, 5, 5, 5);
    varwindow->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    varwindow->hide();

    update_recents();

    // check if we have OVITO and VMD installed and deacivate actions if not
    ui->actionView_in_OVITO->setEnabled(has_exe("ovito"));
    ui->actionView_in_OVITO->setData("ovito");
    ui->actionView_in_VMD->setEnabled(has_exe("vmd"));
    ui->actionView_in_VMD->setData("vmd");

    connect(ui->actionNew, &QAction::triggered, this, &LammpsGui::new_document);
    connect(ui->actionOpen, &QAction::triggered, this, &LammpsGui::open);
    connect(ui->actionSave, &QAction::triggered, this, &LammpsGui::save);
    connect(ui->actionSave_As, &QAction::triggered, this, &LammpsGui::save_as);
    connect(ui->actionQuit, &QAction::triggered, this, &LammpsGui::quit);
    connect(ui->actionCopy, &QAction::triggered, this, &LammpsGui::copy);
    connect(ui->actionCut, &QAction::triggered, this, &LammpsGui::cut);
    connect(ui->actionPaste, &QAction::triggered, this, &LammpsGui::paste);
    connect(ui->actionUndo, &QAction::triggered, this, &LammpsGui::undo);
    connect(ui->actionRedo, &QAction::triggered, this, &LammpsGui::redo);
    connect(ui->actionRun_Buffer, &QAction::triggered, this, &LammpsGui::run_buffer);
    connect(ui->actionRun_File, &QAction::triggered, this, &LammpsGui::run_file);
    connect(ui->actionStop_LAMMPS, &QAction::triggered, this, &LammpsGui::stop_run);
    connect(ui->actionSet_Variables, &QAction::triggered, this, &LammpsGui::edit_variables);
    connect(ui->actionImage, &QAction::triggered, this, &LammpsGui::render_image);
    connect(ui->actionAbout_LAMMPS_GUI, &QAction::triggered, this, &LammpsGui::about);
    connect(ui->action_Help, &QAction::triggered, this, &LammpsGui::help);
    connect(ui->actionLAMMPS_GUI_Howto, &QAction::triggered, this, &LammpsGui::howto);
    connect(ui->actionLAMMPS_Manual, &QAction::triggered, this, &LammpsGui::manual);
    connect(ui->actionPreferences, &QAction::triggered, this, &LammpsGui::preferences);
    connect(ui->actionDefaults, &QAction::triggered, this, &LammpsGui::defaults);
    connect(ui->actionView_in_OVITO, &QAction::triggered, this, &LammpsGui::start_exe);
    connect(ui->actionView_in_VMD, &QAction::triggered, this, &LammpsGui::start_exe);
    connect(ui->actionView_Log_Window, &QAction::triggered, this, &LammpsGui::view_log);
    connect(ui->actionView_Graph_Window, &QAction::triggered, this, &LammpsGui::view_chart);
    connect(ui->actionView_Image_Window, &QAction::triggered, this, &LammpsGui::view_image);
    connect(ui->actionView_Variable_Window, &QAction::triggered, this, &LammpsGui::view_variables);
    connect(ui->action_1, &QAction::triggered, this, &LammpsGui::open_recent);
    connect(ui->action_2, &QAction::triggered, this, &LammpsGui::open_recent);
    connect(ui->action_3, &QAction::triggered, this, &LammpsGui::open_recent);
    connect(ui->action_4, &QAction::triggered, this, &LammpsGui::open_recent);
    connect(ui->action_5, &QAction::triggered, this, &LammpsGui::open_recent);

    connect(ui->textEdit->document(), &QTextDocument::modificationChanged, this,
            &LammpsGui::modified);

#if !QT_CONFIG(clipboard)
    ui->actionCut->setEnabled(false);
    ui->actionCopy->setEnabled(false);
    ui->actionPaste->setEnabled(false);
#endif

    lammpsstatus = new QLabel(QString());
    auto pix     = QPixmap(":/lammps-icon-128x128.png");
    lammpsstatus->setPixmap(pix.scaled(22, 22, Qt::KeepAspectRatio));
    ui->statusbar->addWidget(lammpsstatus);
    lammpsstatus->setToolTip("LAMMPS instance is active");
    lammpsstatus->hide();

    auto *lammpsrun   = new QPushButton(QIcon(":/system-run.png"), "");
    auto *lammpsstop  = new QPushButton(QIcon(":/process-stop.png"), "");
    auto *lammpsimage = new QPushButton(QIcon(":/emblem-photos.png"), "");
    lammpsrun->setToolTip("Run LAMMPS on input");
    lammpsstop->setToolTip("Stop LAMMPS");
    lammpsimage->setToolTip("Create snapshot image");
    ui->statusbar->addWidget(lammpsrun);
    ui->statusbar->addWidget(lammpsstop);
    ui->statusbar->addWidget(lammpsimage);
    connect(lammpsrun, &QPushButton::released, this, &LammpsGui::run_buffer);
    connect(lammpsstop, &QPushButton::released, this, &LammpsGui::stop_run);
    connect(lammpsimage, &QPushButton::released, this, &LammpsGui::render_image);

    status = new QLabel("Ready.");
    status->setFixedWidth(300);
    ui->statusbar->addWidget(status);
    dirstatus = new QLabel(QString(" Directory: ") + current_dir);
    dirstatus->setMinimumWidth(400);
    ui->statusbar->addWidget(dirstatus);
    progress = new QProgressBar();
    progress->setRange(0, 1000);
    progress->setMinimumWidth(400);
    progress->hide();
    dirstatus->show();
    ui->statusbar->addWidget(progress);

    if (filename) {
        open_file(filename);
    } else {
        setWindowTitle(QString("LAMMPS-GUI - *unknown*"));
    }
    resize(settings.value("mainx", "500").toInt(), settings.value("mainy", "320").toInt());
}

LammpsGui::~LammpsGui()
{
    delete ui;
    delete highlighter;
    delete capturer;
    delete status;
    delete logwindow;
    delete imagewindow;
    delete chartwindow;
    delete dirstatus;
    delete varwindow;
}

void LammpsGui::new_document()
{
    current_file.clear();
    ui->textEdit->document()->setPlainText(QString());

    if (lammps.is_running()) {
        stop_run();
        runner->wait();
    }
    lammps.close();
    lammpsstatus->hide();
    setWindowTitle(QString("LAMMPS-GUI - *unknown*"));
}

void LammpsGui::open()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open the file");
    open_file(fileName);
}

void LammpsGui::open_recent()
{
    QAction *act = qobject_cast<QAction *>(sender());
    if (act) open_file(act->data().toString());
}

void LammpsGui::start_exe()
{
    if (!lammps.extract_setting("box_exists")) return;
    QAction *act = qobject_cast<QAction *>(sender());
    if (act) {
        auto exe        = act->data().toString();
        QString datacmd = "write_data '";
        QDir datadir(QDir::tempPath());
        QFile datafile(datadir.absoluteFilePath(current_file + ".data"));
        datacmd += datafile.fileName() + "'";
        if (exe == "vmd") {
            QStringList args;
            QFile vmdfile(datadir.absoluteFilePath("tmp-loader.vmd"));
            vmdfile.open(QIODevice::WriteOnly);
            vmdfile.write("package require topotools\n");
            vmdfile.write("topo readlammpsdata {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write("}\ntopo guessatom lammps data\n");
            vmdfile.write("animate write psf {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".psf}\nanimate write dcd {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".dcd}\nmol delete top\nmol new {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".psf} type psf waitfor all\nmol addfile {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".dcd} type dcd waitfor all\nfile delete {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write("} {");
            vmdfile.write(vmdfile.fileName().toLocal8Bit());
            vmdfile.write("} {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".dcd} {");
            vmdfile.write(datafile.fileName().toLocal8Bit());
            vmdfile.write(".psf}\n");
            vmdfile.close();
            args << "-e" << vmdfile.fileName();
            lammps.command(datacmd.toLocal8Bit());
            auto *vmd = new QProcess(this);
            vmd->start(exe, args);
        }
        if (exe == "ovito") {
            QStringList args;
            args << datafile.fileName();
            lammps.command(datacmd.toLocal8Bit());
            auto *ovito = new QProcess(this);
            ovito->start(exe, args);
        }
    }
}

void LammpsGui::update_recents(const QString &filename)
{
    QSettings settings;
    recent = settings.value("recent").value<QList<QString>>();

    for (int i = 0; i < recent.size(); ++i) {
        QFileInfo fi(recent[i]);
        if (!fi.isReadable()) {
            recent.removeAt(i);
            i = 0;
        }
    }

    if (!filename.isEmpty() && !recent.contains(filename)) recent.prepend(filename);
    if (recent.size() > 5) recent.removeLast();
    settings.setValue("recent", QVariant::fromValue(recent));

    ui->action_1->setVisible(false);
    if ((recent.size() > 0) && !recent[0].isEmpty()) {
        QFileInfo fi(recent[0]);
        ui->action_1->setText(QString("&1. ") + fi.fileName());
        ui->action_1->setData(recent[0]);
        ui->action_1->setVisible(true);
    }
    ui->action_2->setVisible(false);
    if ((recent.size() > 1) && !recent[1].isEmpty()) {
        QFileInfo fi(recent[1]);
        ui->action_2->setText(QString("&2. ") + fi.fileName());
        ui->action_2->setData(recent[1]);
        ui->action_2->setVisible(true);
    }
    ui->action_3->setVisible(false);
    if ((recent.size() > 2) && !recent[2].isEmpty()) {
        QFileInfo fi(recent[2]);
        ui->action_3->setText(QString("&3. ") + fi.fileName());
        ui->action_3->setData(recent[2]);
        ui->action_3->setVisible(true);
    }
    ui->action_4->setVisible(false);
    if ((recent.size() > 3) && !recent[3].isEmpty()) {
        QFileInfo fi(recent[3]);
        ui->action_4->setText(QString("&4. ") + fi.fileName());
        ui->action_4->setData(recent[3]);
        ui->action_4->setVisible(true);
    }
    ui->action_5->setVisible(false);
    if ((recent.size() > 4) && !recent[4].isEmpty()) {
        QFileInfo fi(recent[4]);
        ui->action_5->setText(QString("&5. ") + fi.fileName());
        ui->action_5->setData(recent[4]);
        ui->action_5->setVisible(true);
    }
}

void LammpsGui::update_variables()
{
    const auto doc = ui->textEdit->toPlainText().split('\n');
    QStringList known;
    QRegularExpression indexvar("^\\s*variable\\s+(\\w+)\\s+index\\s+(.*)");
    QRegularExpression anyvar("^\\s*variable\\s+(\\w+)\\s+(\\w+)\\s+(.*)");
    QRegularExpression usevar("(\\$(\\w)|\\${(\\w+)})");
    QRegularExpression refvar("v_(\\w+)");

    // forget previously listed variables
    variables.clear();

    for (const auto &line : doc) {

        // first find variable definitions.
        // index variables are special since they can be overridden from the command line
        auto index = indexvar.match(line);
        auto any   = anyvar.match(line);

        if (index.hasMatch()) {
            if (index.lastCapturedIndex() >= 2) {
                auto name = index.captured(1);
                if (!known.contains(name)) {
                    variables.append(qMakePair(name, index.captured(2)));
                    known.append(name);
                }
            }
        } else if (any.hasMatch()) {
            if (any.lastCapturedIndex() >= 3) {
                auto name = any.captured(1);
                if (!known.contains(name)) known.append(name);
            }
        }

        // now split line into words and search for use of undefined variables
        auto words = line.split(' ');
        for (const auto &word : words) {
            auto use = usevar.match(word);
            auto ref = refvar.match(word);
            if (use.hasMatch()) {
                auto name = use.captured(use.lastCapturedIndex());
                if (!known.contains(name)) {
                    known.append(name);
                    variables.append(qMakePair(name, QString()));
                }
            }
            if (ref.hasMatch()) {
                auto name = ref.captured(use.lastCapturedIndex());
                if (!known.contains(name)) known.append(name);
            }
        }
    }
}

// open file and switch CWD to path of file
void LammpsGui::open_file(const QString &fileName)
{
    if (ui->textEdit->document()->isModified()) {
        QMessageBox msg;
        msg.setWindowTitle("Unsaved Changes");
        msg.setWindowIcon(windowIcon());
        msg.setText(QString("The buffer ") + current_file + " has changes");
        msg.setInformativeText("Do you want to save the file before opening a new file?");
        msg.setIcon(QMessageBox::Question);
        msg.setStandardButtons(QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        int rv = msg.exec();
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel:
                return;
                break;
            case QMessageBox::No: // fallthrough
            default:
                // do nothing
                break;
        }
    }
    ui->textEdit->setHighlight(CodeEditor::NO_HIGHLIGHT, false);

    QFileInfo path(fileName);
    current_file = path.fileName();
    current_dir  = path.absolutePath();
    QFile file(path.absoluteFilePath());

    update_recents(path.absoluteFilePath());

    QDir::setCurrent(current_dir);
    if (!file.open(QIODevice::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, "Warning",
                             "Cannot open file " + path.absoluteFilePath() + ": " +
                                 file.errorString());
        return;
    }
    setWindowTitle(QString("LAMMPS-GUI - " + current_file));
    QTextStream in(&file);
    QString text = in.readAll();
    ui->textEdit->document()->setPlainText(text);
    ui->textEdit->moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);
    ui->textEdit->document()->setModified(false);
    file.close();
    dirstatus->setText(QString(" Directory: ") + current_dir);
    status->setText("Ready.");

    update_variables();
}

void LammpsGui::write_file(const QString &fileName)
{
    QFileInfo path(fileName);
    current_file = path.fileName();
    current_dir  = path.absolutePath();
    QFile file(path.absoluteFilePath());

    if (!file.open(QIODevice::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, "Warning", "Cannot save file: " + file.errorString());
        return;
    }
    setWindowTitle(QString("LAMMPS-GUI - " + current_file));

    update_recents(path.absoluteFilePath());

    QTextStream out(&file);
    QString text = ui->textEdit->toPlainText();
    out << text;
    if (text.back().toLatin1() != '\n') out << "\n"; // add final newline if missing
    file.close();
    dirstatus->setText(QString(" Directory: ") + current_dir);
    ui->textEdit->document()->setModified(false);
}

void LammpsGui::save()
{
    QString fileName = current_file;
    // If we don't have a filename from before, get one.
    if (fileName.isEmpty()) fileName = QFileDialog::getSaveFileName(this, "Save");

    write_file(fileName);
}

void LammpsGui::save_as()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as");
    write_file(fileName);
}

void LammpsGui::quit()
{
    if (lammps.is_running()) {
        stop_run();
        runner->wait();
    }
    lammps.close();
    lammpsstatus->hide();
    lammps.finalize();

    if (ui->textEdit->document()->isModified()) {
        QMessageBox msg;
        msg.setWindowTitle("Unsaved Changes");
        msg.setWindowIcon(windowIcon());
        msg.setText(QString("The buffer ") + current_file + " has changes");
        msg.setInformativeText("Do you want to save the file before exiting?");
        msg.setIcon(QMessageBox::Question);
        msg.setStandardButtons(QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        int rv = msg.exec();
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel:
                return;
                break;
            case QMessageBox::No: // fallthrough
            default:
                // do nothing
                break;
        }
    }

    // store some global settings
    QSettings settings;
    if (!isMaximized()) {
        settings.setValue("mainx", width());
        settings.setValue("mainy", height());
    }
    settings.sync();
    QCoreApplication::quit();
}

void LammpsGui::copy()
{
#if QT_CONFIG(clipboard)
    ui->textEdit->copy();
#endif
}

void LammpsGui::cut()
{
#if QT_CONFIG(clipboard)
    ui->textEdit->cut();
#endif
}

void LammpsGui::paste()
{
#if QT_CONFIG(clipboard)
    ui->textEdit->paste();
#endif
}

void LammpsGui::undo()
{
    ui->textEdit->undo();
}

void LammpsGui::redo()
{
    ui->textEdit->redo();
}

void LammpsGui::stop_run()
{
    lammps.force_timeout();
}

void LammpsGui::logupdate()
{
    double t_elapsed, t_remain, t_total;
    int completed = 1000;

    // estimate completion percentage
    if (lammps.is_running()) {
        t_elapsed = lammps.get_thermo("cpu");
        t_remain  = lammps.get_thermo("cpuremain");
        t_total   = t_elapsed + t_remain + 1.0e-10;
        completed = t_elapsed / t_total * 1000.0;

        int nline = -1;
        void *ptr = lammps.last_thermo("line", 0);
        if (ptr) {
            nline = *((int *)ptr);
            ui->textEdit->setHighlight(nline, false);
        }

        if (varwindow) {
            int nvar = lammps.id_count("variable");
            char buffer[200];
            QString varinfo("\n");
            for (int i = 0; i < nvar; ++i) {
                lammps.variable_info(i, buffer, 200);
                varinfo += buffer;
            }
            if (nvar == 0) varinfo += "  (none)  ";

            varwindow->setText(varinfo);
            varwindow->adjustSize();
        }
    }

    progress->setValue(completed);
    if (logwindow) {
        const auto text = capturer->GetChunk();
        if (text.size() > 0) {
            logwindow->insertPlainText(text.c_str());
            logwindow->moveCursor(QTextCursor::End);
            logwindow->textCursor().deleteChar();
        }
    }

    // extract cached thermo data
    if (chartwindow) {
        // thermo data is not yet valid during setup
        void *ptr = lammps.last_thermo("setup", 0);
        if (ptr && *(int *)ptr) return;

        ptr = lammps.last_thermo("step", 0);
        if (ptr) {
            int step = 0;
            if (lammps.extract_setting("bigint") == 4)
                step = *(int *)ptr;
            else
                step = (int)*(int64_t *)ptr;
            int ncols = *(int *)lammps.last_thermo("num", 0);

            // check if the column assignment has changed
            // if yes, delete charts and start over
            if (chartwindow->num_charts() > 0) {
                int count     = 0;
                bool do_reset = false;
                if (step < chartwindow->get_step()) do_reset = true;
                for (int i = 0, idx = 0; i < ncols; ++i) {
                    QString label = (const char *)lammps.last_thermo("keyword", i);
                    // no need to store the timestep column
                    if (label == "Step") continue;
                    if (!chartwindow->has_title(label, idx)) {
                        do_reset = true;
                    } else {
                        ++count;
                    }
                    ++idx;
                }
                if (chartwindow->num_charts() != count) do_reset = true;
                if (do_reset) chartwindow->reset_charts();
            }

            if (chartwindow->num_charts() == 0) {
                for (int i = 0; i < ncols; ++i) {
                    QString label = (const char *)lammps.last_thermo("keyword", i);
                    // no need to store the timestep column
                    if (label == "Step") continue;
                    chartwindow->add_chart(label, i);
                }
            }

            for (int i = 0; i < ncols; ++i) {
                int datatype = *(int *)lammps.last_thermo("type", i);
                double data;
                if (datatype == 0) // int
                    data = *(int *)lammps.last_thermo("data", i);
                else if (datatype == 2) // double
                    data = *(double *)lammps.last_thermo("data", i);
                else if (datatype == 4) // bigint
                    data = (double)*(int64_t *)lammps.last_thermo("data", i);
                chartwindow->add_data(step, data, i);
            }
        }
    }
}

void LammpsGui::modified()
{
    const QString modflag(" - *modified*");
    auto title = windowTitle().remove(modflag);
    if (ui->textEdit->document()->isModified())
        setWindowTitle(title + modflag);
    else
        setWindowTitle(title);
}

void LammpsGui::run_done()
{
    if (logupdater) logupdater->stop();
    delete logupdater;
    logupdater = nullptr;
    progress->setValue(1000);
    ui->textEdit->setHighlight(CodeEditor::NO_HIGHLIGHT, false);

    capturer->EndCapture();
    auto log = capturer->GetCapture();
    logwindow->insertPlainText(log.c_str());
    logwindow->moveCursor(QTextCursor::End);

    if (chartwindow) {
        void *ptr = lammps.last_thermo("step", 0);
        if (ptr) {
            int step = 0;
            if (lammps.extract_setting("bigint") == 4)
                step = *(int *)ptr;
            else
                step = (int)*(int64_t *)ptr;
            int ncols = *(int *)lammps.last_thermo("num", 0);
            for (int i = 0; i < ncols; ++i) {
                if (chartwindow->num_charts() == 0) {
                    QString label = (const char *)lammps.last_thermo("keyword", i);
                    // no need to store the timestep column
                    if (label == "Step") continue;
                    chartwindow->add_chart(label, i);
                }
                int datatype = *(int *)lammps.last_thermo("type", i);
                double data;
                if (datatype == 0) // int
                    data = *(int *)lammps.last_thermo("data", i);
                else if (datatype == 2) // double
                    data = *(double *)lammps.last_thermo("data", i);
                else if (datatype == 4) // bigint
                    data = (double)*(int64_t *)lammps.last_thermo("data", i);
                chartwindow->add_data(step, data, i);
            }
        }
    }

    bool success         = true;
    constexpr int BUFLEN = 1024;
    char errorbuf[BUFLEN];

    if (lammps.has_error()) {
        lammps.get_last_error_message(errorbuf, BUFLEN);
        success = false;
    }

    int nline = CodeEditor::NO_HIGHLIGHT;
    void *ptr = lammps.last_thermo("line", 0);
    if (ptr) nline = *((int *)ptr);

    if (success) {
        status->setText("Ready.");
    } else {
        status->setText("Failed.");
        ui->textEdit->setHighlight(nline, true);
        QMessageBox::critical(this, "LAMMPS-GUI Error",
                              QString("Error running LAMMPS:\n\n") + errorbuf);
    }
    ui->textEdit->setCursor(nline);
    progress->hide();
    dirstatus->show();
}

void LammpsGui::do_run(bool use_buffer)
{
    if (lammps.is_running()) {
        QMessageBox::warning(this, "LAMMPS GUI Error",
                             "Must stop current run before starting a new run");
        return;
    }

    if (!use_buffer && ui->textEdit->document()->isModified()) {
        QMessageBox msg;
        msg.setWindowTitle("Unsaved Changes");
        msg.setWindowIcon(windowIcon());
        msg.setText(QString("The buffer ") + current_file + " has changes");
        msg.setInformativeText("Do you want to save the buffer before running LAMMPS?");
        msg.setIcon(QMessageBox::Question);
        msg.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
        int rv = msg.exec();
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel: // falthrough
            default:
                return;
                break;
        }
    }

    QSettings settings;
    progress->setValue(0);
    dirstatus->hide();
    progress->show();
    int nthreads = settings.value("nthreads", 1).toInt();
    int accel    = settings.value("accelerator", AcceleratorTab::None).toInt();
    if ((accel != AcceleratorTab::OpenMP) && (accel != AcceleratorTab::Intel) &&
        (accel != AcceleratorTab::Kokkos))
        nthreads = 1;
    if (nthreads > 1)
        status->setText(QString("Running LAMMPS with %1 thread(s)...").arg(nthreads));
    else
        status->setText(QString("Running LAMMPS ..."));
    status->repaint();
    start_lammps();
    if (!lammps.is_open()) return;
    capturer->BeginCapture();

    runner     = new LammpsRunner(this);
    is_running = true;
    if (use_buffer) {
        // always add final newline since the text edit widget does not
        char *input = mystrdup(ui->textEdit->toPlainText().toStdString() + "\n");
        runner->setup_run(&lammps, input, nullptr);
    } else {
        char *fname = mystrdup(current_file.toStdString());
        runner->setup_run(&lammps, nullptr, fname);
    }

    connect(runner, &LammpsRunner::resultReady, this, &LammpsGui::run_done);
    connect(runner, &LammpsRunner::finished, runner, &QObject::deleteLater);
    runner->start();

    // if configured, delete old log window before opening new one
    if (settings.value("logreplace", false).toBool()) delete logwindow;
    logwindow = new LogWindow();
    logwindow->setReadOnly(true);
    logwindow->setCenterOnScroll(true);
    logwindow->moveCursor(QTextCursor::End);
    if (use_buffer)
        logwindow->setWindowTitle("LAMMPS-GUI - Output from running LAMMPS on buffer - " +
                                  current_file);
    else
        logwindow->setWindowTitle("LAMMPS-GUI - Output from running LAMMPS on file - " +
                                  current_file);
    logwindow->setWindowIcon(QIcon(":/lammps-icon-128x128.png"));
    QFont text_font;
    text_font.fromString(settings.value("textfont", text_font.toString()).toString());
    logwindow->document()->setDefaultFont(text_font);
    logwindow->setLineWrapMode(LogWindow::NoWrap);
    logwindow->setMinimumSize(400, 300);
    QShortcut *shortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_W), logwindow);
    QObject::connect(shortcut, &QShortcut::activated, logwindow, &LogWindow::close);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Slash), logwindow);
    QObject::connect(shortcut, &QShortcut::activated, this, &LammpsGui::stop_run);
    if (settings.value("viewlog", true).toBool())
        logwindow->show();
    else
        logwindow->hide();

    // if configured, delete old log window before opening new one
    if (settings.value("chartreplace", false).toBool()) delete chartwindow;
    chartwindow = new ChartWindow(current_file);
    if (use_buffer)
        chartwindow->setWindowTitle("LAMMPS-GUI - Thermo charts from running LAMMPS on buffer - " +
                                    current_file);
    else
        chartwindow->setWindowTitle("LAMMPS-GUI - Thermo charts from running LAMMPS on file - " +
                                    current_file);
    chartwindow->setWindowIcon(QIcon(":/lammps-icon-128x128.png"));
    chartwindow->setMinimumSize(400, 300);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_W), chartwindow);
    QObject::connect(shortcut, &QShortcut::activated, chartwindow, &ChartWindow::close);
    shortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Slash), chartwindow);
    QObject::connect(shortcut, &QShortcut::activated, this, &LammpsGui::stop_run);
    if (settings.value("viewchart", true).toBool())
        chartwindow->show();
    else
        chartwindow->hide();

    logupdater = new QTimer(this);
    connect(logupdater, &QTimer::timeout, this, &LammpsGui::logupdate);
    logupdater->start(100);
}

void LammpsGui::render_image()
{
    // LAMMPS is not re-entrant, so we can only query LAMMPS when it is not running
    if (!lammps.is_running()) {
        start_lammps();
        if (!lammps.extract_setting("box_exists")) {
            QMessageBox::warning(this, "ImageViewer Error",
                                 "Cannot create snapshot image without a system box");
            return;
        }
        // if configured, delete old image window before opening new one
        if (QSettings().value("imagereplace", false).toBool()) delete imagewindow;
        imagewindow = new ImageViewer(current_file, &lammps);
    } else {
        QMessageBox::warning(this, "ImageViewer Error",
                             "Cannot create snapshot image while LAMMPS is running");
        return;
    }
    imagewindow->show();
}

void LammpsGui::view_chart()
{
    QSettings settings;
    if (chartwindow) {
        if (chartwindow->isVisible()) {
            chartwindow->hide();
            settings.setValue("viewchart", false);
        } else {
            chartwindow->show();
            settings.setValue("viewchart", true);
        }
    }
}

void LammpsGui::view_log()
{
    QSettings settings;
    if (logwindow) {
        if (logwindow->isVisible()) {
            logwindow->hide();
            settings.setValue("viewlog", false);
        } else {
            logwindow->show();
            settings.setValue("viewlog", true);
        }
    }
}

void LammpsGui::view_image()
{
    if (imagewindow) {
        if (imagewindow->isVisible()) {
            imagewindow->hide();
        } else {
            imagewindow->show();
        }
    }
}

void LammpsGui::view_variables()
{
    if (varwindow) {
        if (varwindow->isVisible()) {
            varwindow->hide();
        } else {
            varwindow->show();
        }
    }
}

void LammpsGui::about()
{
    std::string version = "This is LAMMPS-GUI version " LAMMPS_GUI_VERSION;
    if (lammps.has_plugin()) {
        version += " - LAMMPS library loaded as plugin";
        if (!plugin_path.empty()) {
            version += " from file ";
            version += plugin_path;
        }
    } else {
        version += " - LAMMPS library linked to executable";
    }

    QString to_clipboard(version.c_str());
    to_clipboard += "\n\n";

    std::string info = "LAMMPS is currently running. LAMMPS config info not available.";

    // LAMMPS is not re-entrant, so we can only query LAMMPS when it is not running
    if (!lammps.is_running()) {
        start_lammps();
        capturer->BeginCapture();
        lammps.command("info config");
        capturer->EndCapture();
        info       = capturer->GetCapture();
        auto start = info.find("LAMMPS version:");
        auto end   = info.find("Info-Info-Info", start);
        info       = std::string(info, start, end - start);
    }

    to_clipboard += info.c_str();
    QGuiApplication::clipboard()->setText(to_clipboard);
    info += "(Note: this text has been copied to the clipboard)\n";

    QMessageBox msg;
    msg.setWindowTitle("About LAMMPS");
    msg.setText(version.c_str());
    msg.setInformativeText(info.c_str());
    msg.setIconPixmap(QPixmap(":/lammps-icon-128x128.png").scaled(64, 64));
    msg.setStandardButtons(QMessageBox::Close);
    QFont font;
    font.setPointSizeF(font.pointSizeF() * 0.75);
    msg.setFont(font);

    auto *minwidth      = new QSpacerItem(700, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);
    QGridLayout *layout = (QGridLayout *)msg.layout();
    layout->addItem(minwidth, layout->rowCount(), 0, 1, layout->columnCount());

    msg.exec();
}

void LammpsGui::help()
{
    QMessageBox msg;
    msg.setWindowTitle("LAMMPS-GUI Quick Help");
    msg.setText("<div>This is LAMMPS-GUI version " LAMMPS_GUI_VERSION "</div>");
    msg.setInformativeText("<p>LAMMPS GUI is a graphical text editor that is linked to the LAMMPS "
                           "library and thus can run LAMMPS directly using the contents of the "
                           "text buffer as input through the LAMMPS C-library interface. </p>"
                           "<p>The main window of the LAMMPS GUI is a text editor window with "
                           "LAMMPS specific syntax highlighting. When typing <b>Ctrl-Enter</b> "
                           "or clicking on 'Run LAMMMPS' in the 'Run' menu, LAMMPS will be run "
                           "with the contents of editor buffer as input. The output of the LAMMPS "
                           "run is captured and displayed in a log window. The thermodynamic data "
                           "is displayed in a chart window. Both are updated regularly during the "
                           "run, as is a progress bar in the main window. The running simulation "
                           "can be stopped cleanly by typing <b>Ctrl-/</b> or by clicking on "
                           "'Stop LAMMPS' in the 'Run' menu. After the simulation is finished "
                           "or stopped, an image of the simulated system can be shown in an image "
                           "viewer window by typing <b>Ctrl-i</b> or by clicking on 'View Image' "
                           "in the 'Run' menu. Multiple image settings can be changed through the "
                           "buttons in the menu bar and the image will be re-renderd.</p>"
                           "<p>When opening a file, the editor will determine the directory "
                           "where the input file resides and switch its current working directory "
                           "to that same folder and thus enabling the run to read other files in "
                           "that folder, e.g. a data file .The GUI will show its current working "
                           "directory in the status bar. In addition to using the menu, the "
                           "editor window can also receive files as the first command line "
                           "argument or via drag-n-drop from a graphical file manager or a "
                           "desktop environment.</p>"
                           "<p>Almost all commands are accessible via hotkeys. Which those "
                           "hotkeys are, is typically shown next to their entries in the menus. "
                           "In addition, the documentation for the command in the current line "
                           "can be viewed by typing <b>Ctrl-?</b> or by choosing the respective "
                           "entry in the context menu, available by right-clicking the mouse. "
                           "Log, chart, and image windows can be closed with <b>Ctrl-W</b>, the "
                           "main window with <b>Ctrl-Q</b>.</p>"
                           "<p>The 'About LAMMPS' dialog will show the LAMMPS version and the "
                           "features included into the LAMMPS library linked to the LAMMPS GUI. "
                           "A number of settings can be adjusted in the 'Preferences' dialog (in "
                           "the 'Edit' menu or from <b>Ctrl-P</b>) which includes selecting "
                           "accelerator packages and number of OpenMP threads. Due to its nature "
                           "as a graphical application, it is <b>not</b> possible to use the "
                           "LAMMPS GUI in parallel with MPI.</p>");
    msg.setIconPixmap(QPixmap(":/lammps-icon-128x128.png").scaled(64, 64));
    msg.setStandardButtons(QMessageBox::Close);
    msg.exec();
}

void LammpsGui::manual()
{
    QDesktopServices::openUrl(QUrl("https://docs.lammps.org/"));
}

void LammpsGui::howto()
{
    QDesktopServices::openUrl(QUrl("https://docs.lammps.org/Howto_lammps_gui.html"));
}

void LammpsGui::defaults()
{
    QSettings settings;
    settings.clear();
    settings.sync();
}

void LammpsGui::edit_variables()
{
    QList<QPair<QString, QString>> newvars = variables;
    SetVariables vars(newvars);
    if (vars.exec() == QDialog::Accepted) {
        variables = newvars;
        if (lammps.is_running()) {
            stop_run();
            runner->wait();
            delete runner;
        }
        lammps.close();
        lammpsstatus->hide();
    }
}

void LammpsGui::preferences()
{
    QSettings settings;
    int oldthreads = settings.value("nthreads", 1).toInt();
    int oldaccel   = settings.value("accelerator", AcceleratorTab::None).toInt();
    bool oldecho   = settings.value("echo", 0).toBool();
    bool oldcite   = settings.value("cite", 0).toBool();

    Preferences prefs(&lammps);
    if (prefs.exec() == QDialog::Accepted) {
        // must delete LAMMPS instance after preferences have changed that require
        // using different command line flags when creating the LAMMPS instance like
        // suffixes or package commands
        if ((oldaccel != settings.value("accelerator", AcceleratorTab::None).toInt()) ||
            (oldthreads != settings.value("nthreads", 1).toInt()) ||
            (oldecho != settings.value("echo", 0).toBool()) ||
            (oldcite != settings.value("cite", 0).toBool())) {
            if (lammps.is_running()) {
                stop_run();
                runner->wait();
                delete runner;
            }
            lammps.close();
            lammpsstatus->hide();
        }
        if (imagewindow) imagewindow->createImage();
    }
}

void LammpsGui::start_lammps()
{
    // temporary extend lammps_args with additional arguments
    int initial_narg = lammps_args.size();
    QSettings settings;
    int nthreads = settings.value("nthreads", 1).toInt();
    int accel    = settings.value("accelerator", AcceleratorTab::None).toInt();
    if (accel == AcceleratorTab::Opt) {
        lammps_args.push_back(mystrdup("-suffix"));
        lammps_args.push_back(mystrdup("opt"));
    } else if (accel == AcceleratorTab::OpenMP) {
        lammps_args.push_back(mystrdup("-suffix"));
        lammps_args.push_back(mystrdup("omp"));
        lammps_args.push_back(mystrdup("-pk"));
        lammps_args.push_back(mystrdup("omp"));
        lammps_args.push_back(mystrdup(std::to_string(nthreads)));
    } else if (accel == AcceleratorTab::Intel) {
        lammps_args.push_back(mystrdup("-suffix"));
        lammps_args.push_back(mystrdup("intel"));
        lammps_args.push_back(mystrdup("-pk"));
        lammps_args.push_back(mystrdup("intel"));
        lammps_args.push_back(mystrdup(std::to_string(nthreads)));
    } else if (accel == AcceleratorTab::Gpu) {
        lammps_args.push_back(mystrdup("-suffix"));
        lammps_args.push_back(mystrdup("gpu"));
        lammps_args.push_back(mystrdup("-pk"));
        lammps_args.push_back(mystrdup("gpu"));
        lammps_args.push_back(mystrdup("0"));
    } else if (accel == AcceleratorTab::Kokkos) {
        lammps_args.push_back(mystrdup("-kokkos"));
        lammps_args.push_back(mystrdup("on"));
        lammps_args.push_back(mystrdup("t"));
        lammps_args.push_back(mystrdup(std::to_string(nthreads)));
        lammps_args.push_back(mystrdup("-suffix"));
        lammps_args.push_back(mystrdup("kk"));
    }
    if (settings.value("echo", "0").toBool()) {
        lammps_args.push_back(mystrdup("-echo"));
        lammps_args.push_back(mystrdup("screen"));
    }
    if (settings.value("cite", "0").toBool()) {
        lammps_args.push_back(mystrdup("-cite"));
        lammps_args.push_back(mystrdup("screen"));
    }

    // add variables, if defined
    for (auto &var : variables) {
        QString name  = var.first;
        QString value = var.second;
        if (!name.isEmpty() && !value.isEmpty()) {
            lammps_args.push_back(mystrdup("-var"));
            lammps_args.push_back(mystrdup(name.toStdString()));
            for (const auto &v : value.split(' '))
                lammps_args.push_back(mystrdup(v.toStdString()));
        }
    }

    char **args = lammps_args.data();
    int narg    = lammps_args.size();
    lammps.open(narg, args);
    lammpsstatus->show();

    // delete additional arguments again (3 were there initially
    while (lammps_args.size() > initial_narg) {
        delete lammps_args.back();
        lammps_args.pop_back();
    }

    if (lammps.has_error()) {
        constexpr int BUFLEN = 1024;
        char errorbuf[BUFLEN];
        lammps.get_last_error_message(errorbuf, BUFLEN);

        QMessageBox::critical(this, "LAMMPS-GUI Error",
                              QString("Error launching LAMMPS:\n\n") + errorbuf);
    }
}

// Local Variables:
// c-basic-offset: 4
// End: