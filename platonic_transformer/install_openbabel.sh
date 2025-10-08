#!/bin/bash

# This script provides the definitive, all-in-one method for compiling Open Babel
# from source in restrictive environments. It works by first compiling the required
# 'coordgen' dependency, then compiling Open Babel against it, and finally
# configuring the user's shell for permanent use.

# Stop the script if any command fails
set -e

# --- 0. Activate the Target Virtual Environment ---
echo "--- Step 0: Locating and activating the virtual environment ---"
REPO_DIR=$(pwd)
VENV_PATH="${REPO_DIR}/.venv"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at '${VENV_PATH}'."
    echo "Please run your './setup.sh' script first to create the environment."
    exit 1
fi

source "${VENV_PATH}/bin/activate"
echo "--- Virtual environment activated successfully. ---"


# --- 1. Configuration and Setup ---
echo "--- Step 1: Setting up build directories ---"
export HOME_DIR=$(eval echo ~)
# Paths for the coordgen dependency
export CG_BUILD_DIR="${HOME_DIR}/coordgen_build"
export CG_INSTALL_DIR="${CG_BUILD_DIR}/install"
# Paths for Open Babel
export OB_BUILD_DIR="${HOME_DIR}/openbabel_build"
export PY_BUILD_DIR="${HOME_DIR}/openbabel_python_build"
export OB_INSTALL_DIR="${OB_BUILD_DIR}/install"

# Clean up all previous build attempts
rm -rf "${CG_BUILD_DIR}" "${OB_BUILD_DIR}" "${PY_BUILD_DIR}"
mkdir -p "${CG_BUILD_DIR}" "${OB_BUILD_DIR}" "${PY_BUILD_DIR}"


# --- 2. Install SWIG Build Dependency ---
echo "--- Step 2: Installing SWIG, a required build tool ---"
uv pip install swig


# --- 3. Compile the 'coordgen' C++ Dependency ---
echo "--- Step 3: Compiling the 'coordgen' dependency ---"
cd "${CG_BUILD_DIR}"
git clone https://github.com/schrodinger/coordgenlibs.git
cd coordgenlibs
mkdir build && cd build

# Configure, compile, and install it to its local directory
cmake .. -DCMAKE_INSTALL_PREFIX="${CG_INSTALL_DIR}"
make -j4
make install
echo "--- 'coordgen' library compiled to ${CG_INSTALL_DIR} ---"


# --- 4. Compile the Open Babel C++ Library ---
echo "--- Step 4: Compiling the Open Babel C++ library ---"
cd "${OB_BUILD_DIR}"
git clone --depth 1 https://github.com/openbabel/openbabel.git
cd openbabel
mkdir build && cd build

# Point cmake to our custom-built coordgen library
export CMAKE_PREFIX_PATH="${CG_INSTALL_DIR}"
cmake .. -DCMAKE_INSTALL_PREFIX="${OB_INSTALL_DIR}"

make -j4
make install
echo "--- Open Babel C++ library compiled to ${OB_INSTALL_DIR} ---"


# --- 5. Download and Manually Patch the Python Wrapper ---
echo "--- Step 5: Downloading and patching the Python wrapper's setup.py ---"
cd "${PY_BUILD_DIR}"
pip download openbabel --no-binary :all:
tar -xvf openbabel-*.tar.gz
cd openbabel-*/

SETUP_PY_FILE="setup.py"
HARDCODED_FUNCTION="def locate_ob():\n    \"\"\"Return the hardcoded paths to the local Open Babel installation.\"\"\"\n    include_dirs = \"${OB_INSTALL_DIR}/include/openbabel3\"\n    library_dirs = \"${OB_INSTALL_DIR}/lib\"\n    print(\"Using hardcoded Open Babel paths:\")\n    print(\"- Include directory:\", include_dirs)\n    print(\"- Library directory:\", library_dirs)\n    return include_dirs, library_dirs"

# Use awk to automatically replace the broken 'locate_ob' function
awk -v replacement="$HARDCODED_FUNCTION" '
  BEGIN {p=1}
  /def locate_ob\(\):/ {print replacement; p=0}
  /return include_dirs, library_dirs/ {p=1; next}
  p {print}
' "$SETUP_PY_FILE" > "${SETUP_PY_FILE}.tmp" && mv "${SETUP_PY_FILE}.tmp" "$SETUP_PY_FILE"
echo "--- setup.py has been patched successfully ---"


# --- 6. Manually Build and Install the Python Wheel ---
echo "--- Step 6: Building the Python wheel manually ---"
python3 setup.py bdist_wheel

echo "--- Step 7: Installing the correctly built wheel into .venv ---"
uv pip install dist/openbabel-*.whl


# --- 8. Make the Runtime Library Path Permanent ---
echo "--- Step 8: Updating shell config to make the change permanent ---"
BASHRC_FILE="${HOME_DIR}/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
   BASHRC_FILE="${HOME_DIR}/.zshrc"
fi

# CRITICAL: The runtime path must include BOTH libraries.
BASHRC_LINE="export LD_LIBRARY_PATH=\"${OB_INSTALL_DIR}/lib:${CG_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH\""

# Add the line to the shell config file only if it's not already there
grep -qF -- "$BASHRC_LINE" "$BASHRC_FILE" || echo -e "\n# Open Babel + Coordgen Local Install\n$BASHRC_LINE" >> "$BASHRC_FILE"
echo "--- ${BASHRC_FILE} has been updated. ---"


echo ""
echo "--- ✅ Open Babel Installation Complete! ---"
echo ""
echo "IMPORTANT: To use Open Babel, you must now either:"
echo ""
echo "  1. Open a NEW terminal window."
echo "     OR"
echo "  2. Run the following command in THIS terminal:"
echo "     source ${BASHRC_FILE}"
echo ""
echo "After that, your setup will be complete."